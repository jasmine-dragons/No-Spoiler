import { useState } from "react";
import "./style.less";

const Comment = ({username, comment, spoiler}) => {
    const [isSpoiler, setSpoiler] = useState(spoiler);
    return (
        <div className="comment">
            <p>{username}</p>
            <h2 className={`content ${spoiler ? "spoiler" : ""}`} onClick={() => setSpoiler(false)}>{comment}</h2>
        </div>
    );
};

export default Comment;