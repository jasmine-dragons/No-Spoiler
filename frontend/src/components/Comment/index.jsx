import "./style.less";

const Comment = ({username, comment, spoiler}) => {
    return (
        <div className="comment">
            <p>{username}</p>
            <h2 className={`content ${spoiler ? "spoiler" : ""}`}>{comment}</h2>
        </div>
    );
};

export default Comment;