import "./style.less"
import { useState } from "react";

const LargePost = ({username, title, body, spoiler}) => {
    const [isSpoiler, setSpoiler] = useState(spoiler);

    return(
        <div className="post-body">
            <p>{username}</p>
            <h1>{title}</h1>
            <h2 className={`content ${spoiler ? "spoiler" : ""}`} onClick={() => setSpoiler(false)}>{body}</h2>
        </div>
    );
};

export default LargePost;