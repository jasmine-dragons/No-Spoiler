import "./style.less";

const Post = ({username, title, selected, onClick}) => {

    return (
        <div className={`post-component ${selected ? "selected" : ""}`} onClick={onClick}>
            <p>{username}</p>
            <h1>{title}</h1>
        </div>
    );
};

export default Post;